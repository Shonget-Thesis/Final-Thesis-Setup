# Test: transmissibility theory of teacher-to-student handover (EXP-026)

Development speakers only, clean prompts; {'diversity_speakers': 20, 'quality_cases': 40} of {'diversity_speakers': 20, 'quality_cases': 40}; preregistration `78f3432d54ebe994…`. Every number is computed from the CSV files by `run_transmissibility_test.py analyze`. This tests a theory's predictions; it is not evidence for a method.

Arms share one teacher trajectory per seed: W4 / W8 = student restarted at tau = 0.0761 from Step 4 / Step 8 (W8 is the 0.25 reference); W4R2 / W8R2 = restarted at tau = 0.2929; T16 = the teacher alone run to the end.

## Verdict: **supported**

- H1, the reference has less pitch variety than the pure teacher: **YES** (T16 - W8 +0.0226 [+0.0089, +0.0447])
- H2, restarting at tau = 0.29 recovers variety: **YES** (W8R2 - W8 +0.0223 [+0.0084, +0.0419]; mean |gap to the teacher|: W8 0.0249, W8R2 0.0148)
- H3, at tau = 0.0761 Step 4 and Step 8 agree (saturation): **YES** (W8 - W4 +0.0007 [-0.0008, +0.0024])
- H4 (exploratory), speakers whose student loses more variety also lose more at the reference: Spearman +0.60

## Pitch variety (CVf0, mean over speakers)

| Arm | Passes | CVf0 |
|---|---:|---:|
| W4 | 13 | 0.0942 |
| W8 | 21 | 0.0948 |
| W4R2 | 12 | 0.1036 |
| W8R2 | 20 | 0.1171 |
| T16 | 32 | 0.1175 |

W4R2 - W4: +0.0094 [-0.0041, +0.0259]; T16 - W8R2: +0.0004 [-0.0108, +0.0099]. Reproduction of EXP-020's stored W4 and W8 CVf0, max |difference|: 0.00e+00.

## Quality of the new arms against the stored reference (40 speakers, speaker bootstrap, 95%)

| Arm | Passes | WER | Similarity | UTMOS |
|---|---:|---|---|---|
| W4R2 | 12 | +0.0001 [-0.0068, +0.0071] | -0.0078 [-0.0157, -0.0010] | -0.427 [-0.588, -0.291] |
| W8R2 | 20 | +0.0018 [+0.0000, +0.0054] | -0.0015 [-0.0065, +0.0034] | -0.169 [-0.310, -0.042] |
| T16 | 32 | -0.0010 [-0.0083, +0.0054] | -0.0131 [-0.0207, -0.0054] | -0.594 [-0.751, -0.458] |

Development evidence only (speakers that designed earlier experiments), clean prompts.
