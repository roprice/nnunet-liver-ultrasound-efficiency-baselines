# Multi-pass detection projected with a beta-binomial model

Prediction snapshot `195716b4a6ec`. nnU-Net default-augmentation final models (588-image AUL training pool, seed 42, 150 epochs, folds 0–4). Values are mean ± sample SD across the five fold models, each fitted separately.

Each patient has a latent per-image detection probability, Beta-distributed across patients. Detection in n passes is the probability that at least one of n images is detected: 1 − B(a, b + n) / B(a, b). The model has two parameters: the mean per-image rate and the within-patient correlation ρ. ρ = 0 means a patient's images behave independently; larger ρ means misses concentrate in particular patients, so extra passes help less. A patient's images are treated as interchangeable draws. Presence uses every Clean image; overlap uses the caliper-derived circular proxies.

## HCC ≤2.5 cm: target projections

Caliper-marked, proxy-eligible images only: 32 patients, 128 images, 26 patients with 2+ images. Presence and overlap use the same images. Circular caliper proxies are approximate lesion references.

The target is a fold-mean detection probability ≥0.99, not ≥0.99 in every fold. Values are mean ± sample SD across five separately fitted fold models, not confidence intervals. Pass counts beyond the observed maximum of 11 images per patient are extrapolations.

Projected detection (mean ± sample SD) is shown for presence and overlap at 1–20 passes.

| passes | presence | overlap (IoU > 0) |
|---:|---|---|
| 1 | 0.837149 ± 0.044020 | 0.712755 ± 0.036295 |
| 2 | 0.937179 ± 0.024725 | 0.856079 ± 0.035455 |
| 3 | 0.966973 ± 0.016919 | 0.910784 ± 0.031012 |
| 4 | 0.979480 ± 0.011978 | 0.937979 ± 0.026799 |
| 5 | 0.985905 ± 0.008772 | 0.953683 ± 0.023321 |
| 6 | 0.989664 ± 0.006645 | 0.963680 ± 0.020513 |
| 7 | 0.992066 ± 0.005189 | 0.970494 ± 0.018239 |
| 8 | 0.993702 ± 0.004159 | 0.975381 ± 0.016375 |
| 9 | 0.994870 ± 0.003407 | 0.979026 ± 0.014829 |
| 10 | 0.995734 ± 0.002843 | 0.981830 ± 0.013530 |
| 11 | 0.996393 ± 0.002411 | 0.984043 ± 0.012427 |
| 12 | 0.996907 ± 0.002071 | 0.985826 ± 0.011479 |
| 13 | 0.997317 ± 0.001800 | 0.987289 ± 0.010657 |
| 14 | 0.997649 ± 0.001580 | 0.988506 ± 0.009939 |
| 15 | 0.997922 ± 0.001400 | 0.989533 ± 0.009306 |
| 16 | 0.998149 ± 0.001249 | 0.990409 ± 0.008745 |
| 17 | 0.998339 ± 0.001122 | 0.991163 ± 0.008244 |
| 18 | 0.998502 ± 0.001015 | 0.991819 ± 0.007795 |
| 19 | 0.998641 ± 0.000922 | 0.992393 ± 0.007389 |
| 20 | 0.998761 ± 0.000843 | 0.992900 ± 0.007022 |

## Fitted parameters

Outcomes are fitted only with at least 20 patients with two or more images.

| outcome | patients | patients with 2+ images | images | per-image rate | fitted mean | ρ |
|---|---:|---:|---:|---|---|---|
| HCC presence, Clean cohort | 421 | 292 | 1389 | 0.894 ± 0.012 | 0.918 ± 0.012 | 0.230 ± 0.038 |
| Hemangioma presence, Clean cohort | 323 | 212 | 1267 | 0.924 ± 0.004 | 0.926 ± 0.002 | 0.063 ± 0.013 |
| Hemangioma overlap (IoU > 0), proxy subset | 190 | 75 | 380 | 0.758 ± 0.017 | 0.754 ± 0.020 | 0.122 ± 0.049 |
| HCC overlap (IoU > 0), caliper subset | 78 | 68 | 306 | 0.731 ± 0.033 | 0.730 ± 0.034 | 0.300 ± 0.053 |
| HCC centroid (d_eq ≤ 1.0), caliper subset | 78 | 68 | 306 | 0.703 ± 0.035 | 0.702 ± 0.036 | 0.310 ± 0.065 |

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
| HCC centroid (d_eq ≤ 1.0), caliper subset | 4 | 26 | 0.985 ± 0.021 | 0.960 ± 0.044 |
| HCC centroid (d_eq ≤ 1.0), caliper subset | 5 | 12 | 0.800 ± 0.075 | 0.972 ± 0.036 |

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
| HCC centroid (d_eq ≤ 1.0), caliper subset | 22 | subsampled | 0.668 ± 0.031 | 0.794 ± 0.051 | 0.838 ± 0.055 | 0.857 ± 0.055 | 0.868 ± 0.052 |
| HCC centroid (d_eq ≤ 1.0), caliper subset | all | model | 0.702 ± 0.036 | 0.845 ± 0.039 | 0.901 ± 0.035 | 0.929 ± 0.031 | 0.946 ± 0.028 |

## Projected detection by number of passes

| passes | HCC presence, Clean cohort | Hemangioma presence, Clean cohort | Hemangioma overlap (IoU > 0), proxy subset | HCC overlap (IoU > 0), caliper subset | HCC centroid (d_eq ≤ 1.0), caliper subset | HCC overlap, transferred ρ |
|---:|---|---|---|---|---|---|
| 1 | 0.918 ± 0.012 | 0.926 ± 0.002 | 0.754 ± 0.020 | 0.730 ± 0.034 | 0.702 ± 0.036 | 0.731 ± 0.033 |
| 2 | 0.976 ± 0.005 | 0.990 ± 0.001 | 0.916 ± 0.020 | 0.867 ± 0.033 | 0.845 ± 0.039 | 0.835 ± 0.059 |
| 3 | 0.990 ± 0.003 | 0.998 ± 0.000 | 0.965 ± 0.015 | 0.918 ± 0.028 | 0.901 ± 0.035 | 0.875 ± 0.076 |
| 4 | 0.995 ± 0.002 | 1.000 ± 0.000 | 0.983 ± 0.011 | 0.943 ± 0.024 | 0.929 ± 0.031 | 0.897 ± 0.084 |
| 5 | 0.997 ± 0.001 | 1.000 ± 0.000 | 0.991 ± 0.007 | 0.958 ± 0.021 | 0.946 ± 0.028 | 0.909 ± 0.089 |
| 6 | 0.998 ± 0.001 | 1.000 ± 0.000 | 0.994 ± 0.005 | 0.967 ± 0.018 | 0.956 ± 0.025 | 0.918 ± 0.091 |
| 7 | 0.999 ± 0.001 | 1.000 ± 0.000 | 0.996 ± 0.004 | 0.973 ± 0.016 | 0.964 ± 0.022 | 0.924 ± 0.093 |
| 8 | 0.999 ± 0.001 | 1.000 ± 0.000 | 0.998 ± 0.003 | 0.978 ± 0.014 | 0.969 ± 0.020 | 0.928 ± 0.094 |
| 9 | 0.999 ± 0.000 | 1.000 ± 0.000 | 0.998 ± 0.002 | 0.981 ± 0.013 | 0.974 ± 0.018 | 0.932 ± 0.095 |
| 10 | 0.999 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.002 | 0.984 ± 0.012 | 0.977 ± 0.017 | 0.935 ± 0.095 |
| 11 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.986 ± 0.011 | 0.979 ± 0.016 | 0.937 ± 0.095 |

The transferred column uses the HCC caliper subset's per-image overlap rate and a borrowed ρ: HCC presence ρ scaled by hemangioma's ratio of overlap ρ to presence ρ (transferred ρ = 0.469 ± 0.255). It assumes that ratio carries over between pathologies.
