# Relative noise-floor sweep

| Fraction | AUL average (px) | Triage | Centroid 0.25 | Centroid 0.5 | Centroid 1.0 | Overlap >0 | Overlap >0.2 | Overlap >0.5 |
|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| 0.00005 | 29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.00010 | 58 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.00020 | 117 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.00030 | 176 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.00040 | 235 | 26 | 11 | 22 | 22 | 22 | 3 | 0 |
| 0.00050 | 294 | 51 | 27 | 42 | 42 | 41 | 12 | 3 |

Each row is a candidate noise floor as a fraction of image area. Values are counts of mass-present test cases detected under the selected floor (0.0003) that are missed under the candidate, per detection criterion.

The AUL-average column converts each fraction to pixels using the mean test-image area of 589,926 px.
