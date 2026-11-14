# Pilot: Decision Handover (EXP-023)

Development speakers only; {'cases': 60, 'seed_pairs': 20, 'diversity_pairs': 10} of {'cases': 60, 'seed_pairs': 20, 'diversity_pairs': 10}; preregistration `469c90edf7c6094d…`. Every number is computed from the CSV files by `run_decision_handover_pilot.py analyze`. A pilot decides whether a frozen test on new speakers is worth running; it cannot support a claim.

Decision Handover hands the student only the teacher's change since its first estimate, x_k = S + slow(T_k - T_0). DHc down-weights the student's own prosody by sqrt(1 - rho_k); DHu lets the teacher decide without guidance.

## Go / no-go

- G1, the teacher's first estimate is shared across seeds: **FAIL**
- G2, a variant keeps quality and pitch variety: DH_2 FAIL, DH_3 FAIL, DHc_2 FAIL, DHc_3 FAIL, DHu_2 FAIL, DHu_3 FAIL
- **Advance to a frozen test: NO**
- G3, the Two-Axis Handover keeps pitch variety: **PASS**
- **Two-Axis Handover advances to a frozen test on new speakers: YES**
- G4, the Agreement-Routed Handover keeps quality, cost and pitch variety: **FAIL**
- **Agreement-Routed Handover advances to a frozen test on new speakers: NO**

## Assumptions (teacher only, 4 seeds per pair)

| Condition | Trajectory | Spread at Step 0 / Step 3, slow part | Whole estimate | Voice part | Shared share of the decision, Steps 1-4 |
|---|---|---:|---:|---:|---|
| clean | guided | 0.146 | 0.127 | 0.122 | 0.71, 0.81, 0.83, 0.82 (independent: 0.50) |
| clean | unguided | 0.240 | 0.216 | 0.261 | 0.61, 0.57, 0.56, 0.56 (independent: 0.50) |
| noisy15 | guided | 0.296 | 0.376 | 0.605 | 0.89, 0.94, 0.93, 0.91 (independent: 0.50) |
| noisy15 | unguided | 0.264 | 0.372 | 0.562 | 0.78, 0.68, 0.62, 0.60 (independent: 0.50) |

## Quality against the 0.25 reference (speaker bootstrap, 95%)

| Output | Passes | WER | Similarity | UTMOS | WER, degraded | WER vs STS, degraded | Failures |
|---|---:|---|---|---|---|---|---:|
| DH_2 | 10 | +0.1164 [+0.0635, +0.1743] | +0.0248 [+0.0027, +0.0475] | +0.606 [+0.397, +0.814] | +0.1640 [+0.0866, +0.2467] | +0.1656 [+0.0713, +0.2668] | 25/60 |
| DH_3 | 12 | +0.1150 [+0.0574, +0.1816] | +0.0317 [+0.0114, +0.0529] | +0.501 [+0.301, +0.701] | +0.1586 [+0.0732, +0.2585] | +0.1673 [+0.0682, +0.2823] | 25/60 |
| DHc_2 | 10 | +0.0624 [+0.0171, +0.1094] | +0.0309 [+0.0098, +0.0535] | +0.793 [+0.614, +0.967] | +0.0864 [+0.0154, +0.1570] | +0.0880 [+0.0255, +0.1609] | 22/60 |
| DHc_3 | 12 | +0.0093 [-0.0435, +0.0562] | +0.0411 [+0.0210, +0.0621] | +0.845 [+0.623, +1.057] | +0.0123 [-0.0652, +0.0821] | +0.0210 [-0.0125, +0.0590] | 12/60 |
| DHu_2 | 7 | -0.0249 [-0.0610, +0.0062] | +0.0420 [+0.0240, +0.0607] | +0.651 [+0.443, +0.861] | -0.0422 [-0.0956, +0.0013] | -0.0405 [-0.0730, -0.0042] | 12/60 |
| DHu_3 | 8 | -0.0132 [-0.0558, +0.0239] | +0.0443 [+0.0284, +0.0612] | +0.620 [+0.412, +0.818] | -0.0234 [-0.0863, +0.0349] | -0.0147 [-0.0636, +0.0343] | 10/60 |
| STT_4 | 14 | -0.0094 [-0.0520, +0.0223] | +0.0493 [+0.0313, +0.0686] | +0.713 [+0.480, +0.931] | -0.0140 [-0.0779, +0.0335] | — | 8/60 |
| student4 | 4 | -0.0319 [-0.0775, +0.0049] | +0.0484 [+0.0310, +0.0653] | +0.745 [+0.546, +0.944] | -0.0465 [-0.1133, +0.0078] | — | 8/60 |
| STS_2 | 10 | -0.0011 [-0.0474, +0.0366] | +0.0353 [+0.0129, +0.0587] | +0.581 [+0.372, +0.790] | -0.0016 [-0.0711, +0.0549] | — | 12/60 |
| STS_3 | 12 | -0.0046 [-0.0526, +0.0341] | +0.0407 [+0.0216, +0.0611] | +0.713 [+0.478, +0.942] | -0.0087 [-0.0792, +0.0497] | — | 12/60 |

## Pitch variety, clean prompts (CVf0; stored reference, STS and student on the same speakers and seeds)

| Output | CVf0 | minus reference | minus STS at the same step | minus student alone |
|---|---:|---|---|---|
| DH_2 | 0.0654 | -0.0288 [-0.0633, -0.0086] | -0.0132 [-0.0276, -0.0037] | +0.0136 [+0.0080, +0.0208] |
| DH_3 | 0.0832 | -0.0110 [-0.0443, +0.0103] | -0.0002 [-0.0161, +0.0131] | +0.0314 [+0.0195, +0.0475] |
| DHc_2 | 0.0708 | -0.0234 [-0.0566, -0.0027] | -0.0078 [-0.0206, +0.0010] | +0.0190 [+0.0084, +0.0342] |
| DHc_3 | 0.0821 | -0.0121 [-0.0331, +0.0008] | -0.0013 [-0.0043, +0.0013] | +0.0303 [+0.0173, +0.0492] |
| DHu_2 | 0.0559 | -0.0383 [-0.0750, -0.0164] | -0.0227 [-0.0401, -0.0103] | +0.0041 [+0.0005, +0.0078] |
| DHu_3 | 0.0672 | -0.0270 [-0.0687, +0.0029] | -0.0162 [-0.0397, +0.0058] | +0.0153 [+0.0066, +0.0280] |

## Two-Axis Handover (second candidate)

Frozen parameters: SNR threshold 26.55 dB, Voice-Swap band 2.0 dB, Step-3 action threshold 1.2154.

| Clean-prompt pitch variety (CVf0 minus the stored reference) | Mean [95% CI] | Speakers |
|---|---|---:|
| Two-Axis Handover (G3) | -0.0013 [-0.0042, +0.0010] | 10 |
| whole estimate at Step 3 | -0.0009 [-0.0040, +0.0018] | 10 |
| whole estimate at Step 4 | -0.0010 [-0.0039, +0.0013] | 10 |

Branches of the diversity prompts: {'whole': 10}; mean share of seeds handed over at Step 3: 0.95; W4 against EXP-020's stored plain_4, max |difference|: 0.0000.

In-sample replay on the 360 stored cases (descriptive; the same data fixed the parameters): 7.16 mean passes (10.80 on clean prompts), routes {'W3': 89, 'W4': 7, 'STT_4': 45, 'student4': 219}; clean WER +0.0007 [-0.0042, +0.0063], similarity +0.0022 [+0.0003, +0.0040], UTMOS -0.045 [-0.098, +0.005]; degraded WER -0.0249 [-0.0407, -0.0106], similarity +0.0730 [+0.0593, +0.0870]; Definition 3.1 failures 0 clean, 48 degraded.

## Agreement-Routed Handover (third candidate)

Out of fold over 5 speaker folds, on this pilot's Step-0 voice gaps (vg_g) and the stored outputs of the same cases; 60 cases, 11.47 mean passes; routes per condition {'clean': {'W4': 19, 'STT_4': 0, 'student4': 1}, 'gauss20': {'W4': 15, 'STT_4': 0, 'student4': 5}, 'noisy15': {'W4': 7, 'STT_4': 0, 'student4': 13}}.

Misroutes (clean case to the student alone, degraded case to the whole estimate): voice gap 23, prompt SNR on the same cases and folds 0 (SNR rule 7.02 mean passes). Absolute quality below is descriptive: 20 speakers cannot resolve it.

| Against the stored reference | WER | Similarity | UTMOS |
|---|---|---|---|
| clean | +0.0000 [+0.0000, +0.0000] | +0.0020 [-0.0020, +0.0061] | -0.026 [-0.110, +0.042] |
| degraded | +0.0129 [-0.0791, +0.1113] | +0.0368 [+0.0114, +0.0627] | +0.377 [+0.159, +0.624] |

Definition 3.1 failures on clean prompts: 0 of 20. Clean-prompt pitch variety (CVf0 minus the stored reference): -0.0210 [-0.0631, +0.0012] over 10 speakers.

## Decision signals (means per condition)

| Condition | dn_g_1 | dn_g_2 | dn_g_3 | dn_u_1 | dn_u_2 | dn_u_3 | dc_g_2 | dc_g_3 | dc_u_2 | dc_u_3 | dr_g_1 | dr_g_2 | dr_g_3 | dr_u_1 | dr_u_2 | dr_u_3 | vg_g | vg_u |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 29.203 | 130.543 | 193.434 | 10.358 | 35.823 | 57.571 | 0.649 | 0.825 | 0.875 | 0.942 | 0.195 | 0.845 | 1.273 | 0.116 | 0.403 | 0.645 | 0.992 | 0.120 |
| gauss20 | 56.587 | 141.492 | 175.618 | 16.938 | 36.725 | 52.320 | 0.856 | 0.950 | 0.889 | 0.932 | 0.381 | 0.917 | 1.158 | 0.232 | 0.510 | 0.722 | 1.678 | 0.158 |
| noisy15 | 67.746 | 141.239 | 164.380 | 19.722 | 38.814 | 53.881 | 0.899 | 0.974 | 0.868 | 0.931 | 0.486 | 0.969 | 1.135 | 0.289 | 0.569 | 0.785 | 1.819 | 0.188 |

Completion weights from EXP-022: rho {2: 0.6390551133435706, 3: 0.9414867952607213}, g {2: 0.6007868895510532, 3: 0.24189502834758445}.

Development evidence only (speakers that designed EXP-020), white-noise prompts. A claim needs a frozen test on new speakers.
