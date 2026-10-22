# Multi-pass detection projected with a beta-binomial model

Prediction snapshot `195716b4a6ec`. nnU-Net default-augmentation final models (588-image AUL training pool, seed 42, 150 epochs, folds 0–4). Values are mean ± sample SD across the five fold models, each fitted separately.

Each patient has a latent per-image detection probability, Beta-distributed across patients. Detection in n passes is the probability that at least one of n images is detected: 1 − B(a, b + n) / B(a, b). The model has two parameters: the mean per-image rate and the within-patient correlation ρ. ρ = 0 means a patient's images behave independently; larger ρ means misses concentrate in particular patients, so extra passes help less. A patient's images are treated as interchangeable draws. Presence uses every Clean image; overlap uses the caliper-derived circular proxies.

## Fitted parameters

Outcomes are fitted only with at least 20 patients with two or more images.

| outcome | patients | patients with 2+ images | images | per-image rate | fitted mean | ρ |
|---|---:|---:|---:|---|---|---|
| HCC presence, Clean cohort | 421 | 292 | 1389 | 0.894 ± 0.012 | 0.918 ± 0.012 | 0.230 ± 0.038 |
| Hemangioma presence, Clean cohort | 323 | 212 | 1267 | 0.924 ± 0.004 | 0.926 ± 0.002 | 0.063 ± 0.013 |
| Hemangioma overlap (IoU > 0), proxy subset | 190 | 75 | 380 | 0.758 ± 0.017 | 0.754 ± 0.020 | 0.122 ± 0.049 |
| HCC overlap (IoU > 0), caliper subset | 78 | 68 | 306 | 0.731 ± 0.033 | 0.730 ± 0.034 | 0.300 ± 0.053 |

## Out-of-sample check

Fitted only on patients with ≤3 images, then used to predict detection for patients with exactly 4 or 5 images, which the fit never saw.

| outcome | views | patients | observed | predicted |
|---|---:|---:|---|---|
| HCC presence, Clean cohort | 4 | 43 | 1.000 ± 0.000 | 0.993 ± 0.003 |
| HCC presence, Clean cohort | 5 | 29 | 1.000 ± 0.000 | 0.996 ± 0.002 |
| Hemangioma presence, Clean cohort | 4 | 21 | 1.000 ± 0.000 | 0.997 ± 0.004 |
| Hemangioma presence, Clean cohort | 5 | 21 | 1.000 ± 0.000 | 0.998 ± 0.002 |
| Hemangioma overlap (IoU > 0), proxy subset | 4 | 9 | 1.000 ± 0.000 | 0.968 ± 0.018 |
| Hemangioma overlap (IoU > 0), proxy subset | 5 | 10 | 1.000 ± 0.000 | 0.979 ± 0.014 |
| HCC overlap (IoU > 0), caliper subset | 4 | 26 | 0.985 ± 0.021 | 0.947 ± 0.054 |
| HCC overlap (IoU > 0), caliper subset | 5 | 12 | 0.850 ± 0.037 | 0.960 ± 0.046 |

## Within-patient subsampling

Patients with ≥5 images. For each, the exact share of all n-image subsets of their own images that contain a detected image, averaged over patients. This compares pass counts within the same patients, unlike the view-count tables, which compare different patients.

| outcome | patients | curve | 1 | 2 | 3 | 4 | 5 |
|---|---:|---|---|---|---|---|---|
| HCC presence, Clean cohort | 83 | subsampled | 0.890 ± 0.015 | 0.966 ± 0.006 | 0.987 ± 0.005 | 0.994 ± 0.004 | 0.997 ± 0.004 |
| HCC presence, Clean cohort | all | model | 0.918 ± 0.012 | 0.976 ± 0.005 | 0.990 ± 0.003 | 0.995 ± 0.002 | 0.997 ± 0.001 |
| Hemangioma presence, Clean cohort | 83 | subsampled | 0.928 ± 0.009 | 0.988 ± 0.005 | 0.995 ± 0.004 | 0.996 ± 0.005 | 0.997 ± 0.005 |
| Hemangioma presence, Clean cohort | all | model | 0.926 ± 0.002 | 0.990 ± 0.001 | 0.998 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Hemangioma overlap (IoU > 0), proxy subset | 19 | subsampled | 0.837 ± 0.023 | 0.970 ± 0.008 | 0.996 ± 0.002 | 0.999 ± 0.000 | 1.000 ± 0.000 |
| Hemangioma overlap (IoU > 0), proxy subset | all | model | 0.754 ± 0.020 | 0.916 ± 0.020 | 0.965 ± 0.015 | 0.983 ± 0.011 | 0.991 ± 0.007 |
| HCC overlap (IoU > 0), caliper subset | 22 | subsampled | 0.711 ± 0.034 | 0.837 ± 0.043 | 0.882 ± 0.039 | 0.903 ± 0.031 | 0.913 ± 0.024 |
| HCC overlap (IoU > 0), caliper subset | all | model | 0.730 ± 0.034 | 0.867 ± 0.033 | 0.918 ± 0.028 | 0.943 ± 0.024 | 0.958 ± 0.021 |

## Projected detection by number of passes

| outcome | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| HCC presence, Clean cohort | 0.918 ± 0.012 | 0.976 ± 0.005 | 0.990 ± 0.003 | 0.995 ± 0.002 | 0.997 ± 0.001 | 0.998 ± 0.001 | 0.999 ± 0.001 | 0.999 ± 0.001 | 0.999 ± 0.000 | 0.999 ± 0.000 |
| Hemangioma presence, Clean cohort | 0.926 ± 0.002 | 0.990 ± 0.001 | 0.998 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Hemangioma overlap (IoU > 0), proxy subset | 0.754 ± 0.020 | 0.916 ± 0.020 | 0.965 ± 0.015 | 0.983 ± 0.011 | 0.991 ± 0.007 | 0.994 ± 0.005 | 0.996 ± 0.004 | 0.998 ± 0.003 | 0.998 ± 0.002 | 0.999 ± 0.002 |
| HCC overlap (IoU > 0), caliper subset | 0.730 ± 0.034 | 0.867 ± 0.033 | 0.918 ± 0.028 | 0.943 ± 0.024 | 0.958 ± 0.021 | 0.967 ± 0.018 | 0.973 ± 0.016 | 0.978 ± 0.014 | 0.981 ± 0.013 | 0.984 ± 0.012 |
| HCC overlap, transferred ρ | 0.731 ± 0.033 | 0.835 ± 0.059 | 0.875 ± 0.076 | 0.897 ± 0.084 | 0.909 ± 0.089 | 0.918 ± 0.091 | 0.924 ± 0.093 | 0.928 ± 0.094 | 0.932 ± 0.095 | 0.935 ± 0.095 |

The transferred row uses the HCC caliper subset's per-image overlap rate and a borrowed ρ: HCC presence ρ scaled by hemangioma's ratio of overlap ρ to presence ρ (transferred ρ = 0.469 ± 0.255). It assumes that ratio carries over between pathologies.
