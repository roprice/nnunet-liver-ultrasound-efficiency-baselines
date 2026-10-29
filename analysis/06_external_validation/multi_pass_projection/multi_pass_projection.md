# Multi-pass detection projected with a beta-binomial model

Prediction snapshot `eb8bd31e22f0`. nnU-Net default-augmentation final models (588-image AUL training pool, seed 42, 150 epochs, folds 0–4). Values are mean ± sample SD across the five fold models, each fitted separately.

Each patient has a latent per-image detection probability, Beta-distributed across patients. Detection in n passes is the probability that at least one of n images is detected: 1 − B(a, b + n) / B(a, b). The model has two parameters: the mean per-image rate and the within-patient correlation ρ. ρ = 0 means a patient's images behave independently; larger ρ means misses concentrate in particular patients, so extra passes help less. A patient's images are treated as interchangeable draws. Presence uses every Clean image; overlap uses the caliper-derived circular proxies.

## HCC ≤2.0 cm: target projections

Exploratory analysis only, not a study finding; fold-to-fold SD does not quantify patient-sampling uncertainty.

Clean HCC images with recorded patient tumor diameter ≤2.0 cm: 90 patients, 331 images, 64 patients with 2+ images. Size comes from the HCC patient table; no proxy-mask eligibility restriction is applied.

The target is a fold-mean detection probability ≥0.99, not ≥0.99 in every fold. Values are mean ± sample SD across five fold models, not confidence intervals. Pass counts beyond the observed maximum of 28 images per patient are extrapolations.

Observed and projected presence are shown at 1–20 passes. Actual patients have exactly the listed number of eligible images; actual images is their total. Observed presence is the fraction of those patients with at least one detected image; NA means no patients had that image count. Projected presence uses all patients in each fold's fit.

| passes | actual patients | actual images | observed presence | projected presence |
|---:|---:|---:|---|---|
| 1 | 26 | 26 | 0.869231 ± 0.043853 | 0.896807 ± 0.019589 |
| 2 | 21 | 42 | 1.000000 ± 0.000000 | 0.966985 ± 0.007205 |
| 3 | 14 | 42 | 1.000000 ± 0.000000 | 0.984829 ± 0.005435 |
| 4 | 8 | 32 | 1.000000 ± 0.000000 | 0.991526 ± 0.004206 |
| 5 | 9 | 45 | 1.000000 ± 0.000000 | 0.994655 ± 0.003269 |
| 6 | 2 | 12 | 1.000000 ± 0.000000 | 0.996340 ± 0.002579 |
| 7 | 1 | 7 | 1.000000 ± 0.000000 | 0.997343 ± 0.002072 |
| 8 | 1 | 8 | 1.000000 ± 0.000000 | 0.997985 ± 0.001694 |
| 9 | 1 | 9 | 1.000000 ± 0.000000 | 0.998420 ± 0.001407 |
| 10 | 1 | 10 | 1.000000 ± 0.000000 | 0.998727 ± 0.001186 |
| 11 | 0 | 0 | NA | 0.998953 ± 0.001012 |
| 12 | 1 | 12 | 1.000000 ± 0.000000 | 0.999124 ± 0.000873 |
| 13 | 2 | 26 | 1.000000 ± 0.000000 | 0.999255 ± 0.000761 |
| 14 | 0 | 0 | NA | 0.999359 ± 0.000668 |
| 15 | 1 | 15 | 1.000000 ± 0.000000 | 0.999443 ± 0.000592 |
| 16 | 0 | 0 | NA | 0.999511 ± 0.000528 |
| 17 | 1 | 17 | 1.000000 ± 0.000000 | 0.999567 ± 0.000473 |
| 18 | 0 | 0 | NA | 0.999615 ± 0.000427 |
| 19 | 0 | 0 | NA | 0.999654 ± 0.000387 |
| 20 | 0 | 0 | NA | 0.999688 ± 0.000353 |

## HCC ≤2.5 cm: target projections

Clean HCC images with recorded patient tumor diameter ≤2.5 cm: 169 patients, 597 images, 118 patients with 2+ images. Size comes from the HCC patient table; no proxy-mask eligibility restriction is applied.

The target is a fold-mean detection probability ≥0.99, not ≥0.99 in every fold. Values are mean ± sample SD across five fold models, not confidence intervals. Pass counts beyond the observed maximum of 28 images per patient are extrapolations.

Observed and projected presence are shown at 1–20 passes. Actual patients have exactly the listed number of eligible images; actual images is their total. Observed presence is the fraction of those patients with at least one detected image; NA means no patients had that image count. Projected presence uses all patients in each fold's fit.

| passes | actual patients | actual images | observed presence | projected presence |
|---:|---:|---:|---|---|
| 1 | 51 | 51 | 0.874510 ± 0.010740 | 0.903580 ± 0.014890 |
| 2 | 41 | 82 | 1.000000 ± 0.000000 | 0.968522 ± 0.003395 |
| 3 | 28 | 84 | 0.971429 ± 0.015972 | 0.985381 ± 0.001869 |
| 4 | 14 | 56 | 1.000000 ± 0.000000 | 0.991850 ± 0.001499 |
| 5 | 12 | 60 | 1.000000 ± 0.000000 | 0.994913 ± 0.001231 |
| 6 | 2 | 12 | 1.000000 ± 0.000000 | 0.996571 ± 0.001011 |
| 7 | 3 | 21 | 1.000000 ± 0.000000 | 0.997556 ± 0.000836 |
| 8 | 2 | 16 | 1.000000 ± 0.000000 | 0.998182 ± 0.000697 |
| 9 | 3 | 27 | 1.000000 ± 0.000000 | 0.998603 ± 0.000587 |
| 10 | 4 | 40 | 1.000000 ± 0.000000 | 0.998897 ± 0.000500 |
| 11 | 1 | 11 | 1.000000 ± 0.000000 | 0.999110 ± 0.000430 |
| 12 | 1 | 12 | 1.000000 ± 0.000000 | 0.999269 ± 0.000373 |
| 13 | 2 | 26 | 1.000000 ± 0.000000 | 0.999390 ± 0.000326 |
| 14 | 0 | 0 | NA | 0.999484 ± 0.000287 |
| 15 | 1 | 15 | 1.000000 ± 0.000000 | 0.999559 ± 0.000255 |
| 16 | 1 | 16 | 1.000000 ± 0.000000 | 0.999619 ± 0.000227 |
| 17 | 1 | 17 | 1.000000 ± 0.000000 | 0.999668 ± 0.000204 |
| 18 | 0 | 0 | NA | 0.999708 ± 0.000184 |
| 19 | 0 | 0 | NA | 0.999742 ± 0.000167 |
| 20 | 0 | 0 | NA | 0.999770 ± 0.000152 |

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
