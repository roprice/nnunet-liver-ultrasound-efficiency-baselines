# External-validation patient detection rates

SMC-LUD Clean: 744 patients, 2656 images per fold model. Canonical dataset snapshot `343555a0eca1` (schema 4; checked against its build manifest).

Patient/size input: `archives/06_external_validation/smc_lud_prepared/smc_lud_manifest.json` (schema 1); SHA-256 `379d7a561b0d0840526bf6cacfe8b43c3c8e280c1086dac95612c8c9f185a7f3`.

Final checkpoints, 588-image AUL training pool, seed 42, 150 epochs, PlainConvUNet. Every model/fold must match the entire clean manifest, including image pathology, dimensions and original-source suffix. Manifest IDs, counts and within-patient pathology/class/diameter consistency are validated. Patients come from the exact image_id → manifest case join, never from parsing image IDs or joining a patient table. Hem_87 remains included despite its missing patient-table row.

For each model, fold and patient, n is the number of selected still images and k is the number with canonical triage_detection_flag=True (any retained predicted mass component; 8-connectivity, area ≥ int(0.0003 × image area)). The **primary patient-weighted flagged fraction is mean(k/n)**, with equal patient weight. It is not the archived runner's strict-majority `patient_weighted_rate`.

Secondary outcomes: image-weighted sum(k)/sum(n), any-image patient rate (k>0), and all-image patient rate (k=n). Patients partition into wholly unflagged (k=0), mixed (0<k<n), and fully flagged (k=n). Miss concentration is the fraction of all unflagged images belonging to wholly unflagged patients; n/a if there are no missed images (a summary is n/a if any fold has that zero denominator).

Each outcome is computed separately in folds 0–4, then summarized as mean ± sample SD across the five fold models. Fold outcomes are never pooled. SD describes model variability on the same patients, not a confidence interval.

These are selected stills from a mass-positive dataset: flags establish neither clinical cancer classification nor confirmed lesion localization. Hemangiomas are positive mass cases; there are no normal controls and specificity is not estimable. Any-image detection has more opportunities to rise with view count, while all-image detection becomes more demanding; view-bin comparisons are descriptive. There are no temporal or visit data, so longitudinal monitoring is not established. HCC size is the patient-table maximum diameter carried in the manifest, not a measurement of every frame. The <2 cm boundary is strict, missing sizes stay separate, and hemangioma size is unavailable.

## Cohort and view counts

| cohort | patients | images | median views | min | max | singletons |
|---|---:|---:|---:|---:|---:|---:|
| Combined | 744 | 2656 | 2 | 1 | 46 | 240 |
| HCC | 421 | 1389 | 2 | 1 | 28 | 129 |
| Hemangioma | 323 | 1267 | 2 | 1 | 46 | 111 |
| HCC <2 cm | 85 | 316 | 2 | 1 | 28 | 25 |
| HCC ≥2 cm | 319 | 1009 | 2 | 1 | 23 | 100 |
| HCC missing size | 17 | 64 | 3 | 1 | 9 | 4 |

Largest image contribution: Hem_511 (46 images).

## Model summaries

Rates are proportions, mean ± sample SD across five fold models (six decimals).

| cohort | model | primary patient-weighted fraction | image-weighted | any-image | all-image | miss concentration |
|---|---|---:|---:|---:|---:|---:|
| Combined | Data efficiency | 0.924148 ± 0.004666 | 0.908283 ± 0.006616 | 0.969086 ± 0.001646 | 0.823387 ± 0.013258 | 0.117057 ± 0.023363 |
| Combined | Brightness augmentation | 0.919613 ± 0.014289 | 0.903916 ± 0.010574 | 0.965860 ± 0.007872 | 0.817204 ± 0.027676 | 0.133362 ± 0.028397 |
| Combined | Gaussian-noise augmentation | 0.922010 ± 0.015046 | 0.906401 ± 0.013281 | 0.968280 ± 0.008042 | 0.821237 ± 0.027265 | 0.116652 ± 0.024833 |
| HCC | Data efficiency | 0.923370 ± 0.012332 | 0.894024 ± 0.011746 | 0.969596 ± 0.005417 | 0.837530 ± 0.024083 | 0.105912 ± 0.018041 |
| HCC | Brightness augmentation | 0.917601 ± 0.017908 | 0.889129 ± 0.013029 | 0.966271 ± 0.011441 | 0.828504 ± 0.030437 | 0.123485 ± 0.042591 |
| HCC | Gaussian-noise augmentation | 0.917648 ± 0.012123 | 0.891145 ± 0.012275 | 0.965796 ± 0.008330 | 0.831829 ± 0.011805 | 0.115169 ± 0.016101 |
| Hemangioma | Data efficiency | 0.925162 ± 0.006636 | 0.923915 ± 0.003512 | 0.968421 ± 0.008866 | 0.804954 ± 0.004895 | 0.130868 ± 0.050856 |
| Hemangioma | Brightness augmentation | 0.922236 ± 0.016716 | 0.920126 ± 0.015114 | 0.965325 ± 0.010590 | 0.802477 ± 0.031145 | 0.142865 ± 0.049549 |
| Hemangioma | Gaussian-noise augmentation | 0.927695 ± 0.022303 | 0.923125 ± 0.018947 | 0.971517 ± 0.009642 | 0.807430 ± 0.051032 | 0.117901 ± 0.042529 |
| HCC <2 cm | Data efficiency | 0.907949 ± 0.013178 | 0.846835 ± 0.019066 | 0.967059 ± 0.009843 | 0.804706 ± 0.044489 | 0.058255 ± 0.016758 |
| HCC <2 cm | Brightness augmentation | 0.899969 ± 0.023266 | 0.831646 ± 0.014742 | 0.964706 ± 0.016638 | 0.795294 ± 0.037757 | 0.058226 ± 0.027930 |
| HCC <2 cm | Gaussian-noise augmentation | 0.903939 ± 0.019706 | 0.838608 ± 0.022265 | 0.967059 ± 0.015339 | 0.807059 ± 0.013414 | 0.053190 ± 0.020152 |
| HCC ≥2 cm | Data efficiency | 0.930166 ± 0.013230 | 0.910406 ± 0.011485 | 0.969906 ± 0.007217 | 0.855799 ± 0.020794 | 0.132347 ± 0.021233 |
| HCC ≥2 cm | Brightness augmentation | 0.923606 ± 0.018931 | 0.906244 ± 0.015758 | 0.966144 ± 0.013001 | 0.843260 ± 0.030635 | 0.160183 ± 0.057574 |
| HCC ≥2 cm | Gaussian-noise augmentation | 0.923917 ± 0.011346 | 0.908226 ± 0.010218 | 0.966144 ± 0.007806 | 0.847649 ± 0.014637 | 0.145822 ± 0.015019 |
| HCC missing size | Data efficiency | 0.872960 ± 0.033740 | 0.868750 ± 0.026146 | 0.976471 ± 0.032219 | 0.658824 ± 0.026307 | 0.080808 ± 0.111569 |
| HCC missing size | Brightness augmentation | 0.893072 ± 0.054274 | 0.903125 ± 0.033874 | 0.976471 ± 0.032219 | 0.717647 ± 0.076696 | 0.072222 ± 0.110414 |
| HCC missing size | Gaussian-noise augmentation | 0.868562 ± 0.037524 | 0.881250 ± 0.026146 | 0.952941 ± 0.049215 | 0.658824 ± 0.049215 | 0.148333 ± 0.156835 |

## View-count effects: combined cohort

Cells show any-image / all-image rates, each mean ± sample SD (three decimals).

| images per patient | patients | images | Data efficiency | Brightness augmentation | Gaussian-noise augmentation |
|---|---:|---:|---|---|---|
| 1 | 240 | 240 | 0.919 ± 0.010 / 0.919 ± 0.010 | 0.917 ± 0.016 / 0.917 ± 0.016 | 0.917 ± 0.021 / 0.917 ± 0.021 |
| 2 | 173 | 346 | 0.985 ± 0.013 / 0.876 ± 0.019 | 0.979 ± 0.018 / 0.873 ± 0.043 | 0.986 ± 0.009 / 0.879 ± 0.040 |
| 3–5 | 215 | 809 | 0.996 ± 0.002 / 0.814 ± 0.029 | 0.993 ± 0.004 / 0.793 ± 0.036 | 0.996 ± 0.004 / 0.801 ± 0.029 |
| 6+ | 116 | 1261 | 0.998 ± 0.004 / 0.564 ± 0.033 | 0.997 ± 0.005 / 0.574 ± 0.042 | 0.997 ± 0.005 / 0.576 ± 0.050 |

## Per-fold audit

Image and patient numerators/denominators are exact. W/M/F = wholly unflagged / mixed / fully flagged patients; these sum to the patient denominator. The last column is missed images in wholly unflagged patients / all missed images.

| cohort | model | fold | flagged images | primary patient-weighted fraction | any-image patients | all-image patients | W/M/F | missed-image concentration counts |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Combined | Data efficiency | 0 | 2426/2656 | 0.928097 | 721/744 | 622/744 | 23/99/622 | 30/230 |
| Combined | Data efficiency | 1 | 2390/2656 | 0.921135 | 723/744 | 600/744 | 21/123/600 | 22/266 |
| Combined | Data efficiency | 2 | 2406/2656 | 0.920897 | 720/744 | 611/744 | 24/109/611 | 27/250 |
| Combined | Data efficiency | 3 | 2406/2656 | 0.920346 | 721/744 | 607/744 | 23/114/607 | 30/250 |
| Combined | Data efficiency | 4 | 2434/2656 | 0.930264 | 720/744 | 623/744 | 24/97/623 | 32/222 |
| Combined | Brightness augmentation | 0 | 2413/2656 | 0.921493 | 720/744 | 613/744 | 24/107/613 | 31/243 |
| Combined | Brightness augmentation | 1 | 2432/2656 | 0.935597 | 725/744 | 627/744 | 19/98/627 | 23/224 |
| Combined | Brightness augmentation | 2 | 2386/2656 | 0.912135 | 712/744 | 603/744 | 32/109/603 | 45/270 |
| Combined | Brightness augmentation | 3 | 2360/2656 | 0.899458 | 713/744 | 575/744 | 31/138/575 | 47/296 |
| Combined | Brightness augmentation | 4 | 2413/2656 | 0.929383 | 723/744 | 622/744 | 21/101/622 | 27/243 |
| Combined | Gaussian-noise augmentation | 0 | 2377/2656 | 0.912886 | 719/744 | 595/744 | 25/124/595 | 33/279 |
| Combined | Gaussian-noise augmentation | 1 | 2464/2656 | 0.944762 | 729/744 | 644/744 | 15/85/644 | 16/192 |
| Combined | Gaussian-noise augmentation | 2 | 2389/2656 | 0.908793 | 713/744 | 600/744 | 31/113/600 | 38/267 |
| Combined | Gaussian-noise augmentation | 3 | 2388/2656 | 0.913747 | 718/744 | 599/744 | 26/119/599 | 37/268 |
| Combined | Gaussian-noise augmentation | 4 | 2419/2656 | 0.929863 | 723/744 | 617/744 | 21/106/617 | 24/237 |
| HCC | Data efficiency | 0 | 1250/1389 | 0.929168 | 408/421 | 360/421 | 13/48/360 | 17/139 |
| HCC | Data efficiency | 1 | 1225/1389 | 0.914365 | 408/421 | 340/421 | 13/68/340 | 14/164 |
| HCC | Data efficiency | 2 | 1232/1389 | 0.919708 | 407/421 | 352/421 | 14/55/352 | 17/157 |
| HCC | Data efficiency | 3 | 1236/1389 | 0.911695 | 406/421 | 346/421 | 15/60/346 | 19/153 |
| HCC | Data efficiency | 4 | 1266/1389 | 0.941914 | 412/421 | 365/421 | 9/47/365 | 11/123 |
| HCC | Brightness augmentation | 0 | 1232/1389 | 0.916292 | 407/421 | 349/421 | 14/58/349 | 19/157 |
| HCC | Brightness augmentation | 1 | 1243/1389 | 0.927075 | 409/421 | 355/421 | 12/54/355 | 16/146 |
| HCC | Brightness augmentation | 2 | 1244/1389 | 0.919821 | 405/421 | 353/421 | 16/52/353 | 18/145 |
| HCC | Brightness augmentation | 3 | 1205/1389 | 0.888612 | 400/421 | 327/421 | 21/73/327 | 35/184 |
| HCC | Brightness augmentation | 4 | 1251/1389 | 0.936204 | 413/421 | 360/421 | 8/53/360 | 10/138 |
| HCC | Gaussian-noise augmentation | 0 | 1223/1389 | 0.913747 | 406/421 | 351/421 | 15/55/351 | 21/166 |
| HCC | Gaussian-noise augmentation | 1 | 1257/1389 | 0.930845 | 410/421 | 357/421 | 11/53/357 | 12/132 |
| HCC | Gaussian-noise augmentation | 2 | 1217/1389 | 0.899467 | 401/421 | 344/421 | 20/57/344 | 22/172 |
| HCC | Gaussian-noise augmentation | 3 | 1244/1389 | 0.918321 | 407/421 | 347/421 | 14/60/347 | 18/145 |
| HCC | Gaussian-noise augmentation | 4 | 1248/1389 | 0.925862 | 409/421 | 352/421 | 12/57/352 | 15/141 |
| Hemangioma | Data efficiency | 0 | 1176/1267 | 0.926702 | 313/323 | 262/323 | 10/51/262 | 13/91 |
| Hemangioma | Data efficiency | 1 | 1165/1267 | 0.929959 | 315/323 | 260/323 | 8/55/260 | 8/102 |
| Hemangioma | Data efficiency | 2 | 1174/1267 | 0.922447 | 313/323 | 259/323 | 10/54/259 | 10/93 |
| Hemangioma | Data efficiency | 3 | 1170/1267 | 0.931622 | 315/323 | 261/323 | 8/54/261 | 11/97 |
| Hemangioma | Data efficiency | 4 | 1168/1267 | 0.915079 | 308/323 | 258/323 | 15/50/258 | 21/99 |
| Hemangioma | Brightness augmentation | 0 | 1181/1267 | 0.928273 | 313/323 | 264/323 | 10/49/264 | 12/86 |
| Hemangioma | Brightness augmentation | 1 | 1189/1267 | 0.946706 | 316/323 | 272/323 | 7/44/272 | 7/78 |
| Hemangioma | Brightness augmentation | 2 | 1142/1267 | 0.902117 | 307/323 | 250/323 | 16/57/250 | 27/125 |
| Hemangioma | Brightness augmentation | 3 | 1155/1267 | 0.913594 | 313/323 | 248/323 | 10/65/248 | 12/112 |
| Hemangioma | Brightness augmentation | 4 | 1162/1267 | 0.920491 | 310/323 | 262/323 | 13/48/262 | 17/105 |
| Hemangioma | Gaussian-noise augmentation | 0 | 1154/1267 | 0.911764 | 313/323 | 244/323 | 10/69/244 | 12/113 |
| Hemangioma | Gaussian-noise augmentation | 1 | 1207/1267 | 0.962901 | 319/323 | 287/323 | 4/32/287 | 4/60 |
| Hemangioma | Gaussian-noise augmentation | 2 | 1172/1267 | 0.920949 | 312/323 | 256/323 | 11/56/256 | 16/95 |
| Hemangioma | Gaussian-noise augmentation | 3 | 1144/1267 | 0.907785 | 311/323 | 252/323 | 12/59/252 | 19/123 |
| Hemangioma | Gaussian-noise augmentation | 4 | 1171/1267 | 0.935078 | 314/323 | 265/323 | 9/49/265 | 9/96 |
| HCC <2 cm | Data efficiency | 0 | 269/316 | 0.912825 | 82/85 | 70/85 | 3/12/70 | 3/47 |
| HCC <2 cm | Data efficiency | 1 | 264/316 | 0.901439 | 83/85 | 65/85 | 2/18/65 | 2/52 |
| HCC <2 cm | Data efficiency | 2 | 260/316 | 0.894131 | 81/85 | 68/85 | 4/13/68 | 4/56 |
| HCC <2 cm | Data efficiency | 3 | 269/316 | 0.903062 | 83/85 | 65/85 | 2/18/65 | 2/47 |
| HCC <2 cm | Data efficiency | 4 | 276/316 | 0.928288 | 82/85 | 74/85 | 3/8/74 | 3/40 |
| HCC <2 cm | Brightness augmentation | 0 | 256/316 | 0.872013 | 80/85 | 65/85 | 5/15/65 | 6/60 |
| HCC <2 cm | Brightness augmentation | 1 | 262/316 | 0.890971 | 82/85 | 66/85 | 3/16/66 | 3/54 |
| HCC <2 cm | Brightness augmentation | 2 | 263/316 | 0.907089 | 82/85 | 70/85 | 3/12/70 | 3/53 |
| HCC <2 cm | Brightness augmentation | 3 | 264/316 | 0.894798 | 82/85 | 65/85 | 3/17/65 | 3/52 |
| HCC <2 cm | Brightness augmentation | 4 | 269/316 | 0.934974 | 84/85 | 72/85 | 1/12/72 | 1/47 |
| HCC <2 cm | Gaussian-noise augmentation | 0 | 259/316 | 0.888223 | 81/85 | 69/85 | 4/12/69 | 4/57 |
| HCC <2 cm | Gaussian-noise augmentation | 1 | 269/316 | 0.910263 | 83/85 | 68/85 | 2/15/68 | 2/47 |
| HCC <2 cm | Gaussian-noise augmentation | 2 | 256/316 | 0.881226 | 81/85 | 67/85 | 4/14/67 | 4/60 |
| HCC <2 cm | Gaussian-noise augmentation | 3 | 269/316 | 0.909148 | 82/85 | 69/85 | 3/13/69 | 3/47 |
| HCC <2 cm | Gaussian-noise augmentation | 4 | 272/316 | 0.930834 | 84/85 | 70/85 | 1/14/70 | 1/44 |
| HCC ≥2 cm | Data efficiency | 0 | 928/1009 | 0.938752 | 310/319 | 279/319 | 9/31/279 | 12/81 |
| HCC ≥2 cm | Data efficiency | 1 | 905/1009 | 0.918779 | 308/319 | 264/319 | 11/44/264 | 12/104 |
| HCC ≥2 cm | Data efficiency | 2 | 915/1009 | 0.927243 | 309/319 | 272/319 | 10/37/272 | 13/94 |
| HCC ≥2 cm | Data efficiency | 3 | 912/1009 | 0.917667 | 307/319 | 270/319 | 12/37/270 | 15/97 |
| HCC ≥2 cm | Data efficiency | 4 | 933/1009 | 0.948387 | 313/319 | 280/319 | 6/33/280 | 8/76 |
| HCC ≥2 cm | Brightness augmentation | 0 | 920/1009 | 0.930717 | 311/319 | 271/319 | 8/40/271 | 11/89 |
| HCC ≥2 cm | Brightness augmentation | 1 | 922/1009 | 0.935961 | 310/319 | 276/319 | 9/34/276 | 13/87 |
| HCC ≥2 cm | Brightness augmentation | 2 | 922/1009 | 0.923312 | 306/319 | 271/319 | 13/35/271 | 15/87 |
| HCC ≥2 cm | Brightness augmentation | 3 | 886/1009 | 0.891146 | 302/319 | 252/319 | 17/50/252 | 31/123 |
| HCC ≥2 cm | Brightness augmentation | 4 | 922/1009 | 0.936894 | 312/319 | 275/319 | 7/37/275 | 9/87 |
| HCC ≥2 cm | Gaussian-noise augmentation | 0 | 910/1009 | 0.924433 | 309/319 | 271/319 | 10/38/271 | 15/99 |
| HCC ≥2 cm | Gaussian-noise augmentation | 1 | 930/1009 | 0.937015 | 310/319 | 277/319 | 9/33/277 | 10/79 |
| HCC ≥2 cm | Gaussian-noise augmentation | 2 | 903/1009 | 0.905692 | 304/319 | 265/319 | 15/39/265 | 17/106 |
| HCC ≥2 cm | Gaussian-noise augmentation | 3 | 919/1009 | 0.925835 | 310/319 | 267/319 | 9/43/267 | 12/90 |
| HCC ≥2 cm | Gaussian-noise augmentation | 4 | 920/1009 | 0.926611 | 308/319 | 272/319 | 11/36/272 | 14/89 |
| HCC missing size | Data efficiency | 0 | 53/64 | 0.831046 | 16/17 | 11/17 | 1/5/11 | 2/11 |
| HCC missing size | Data efficiency | 1 | 56/64 | 0.896172 | 17/17 | 11/17 | 0/6/11 | 0/8 |
| HCC missing size | Data efficiency | 2 | 57/64 | 0.906209 | 17/17 | 12/17 | 0/5/12 | 0/7 |
| HCC missing size | Data efficiency | 3 | 55/64 | 0.842810 | 16/17 | 11/17 | 1/5/11 | 2/9 |
| HCC missing size | Data efficiency | 4 | 57/64 | 0.888562 | 17/17 | 11/17 | 0/6/11 | 0/7 |
| HCC missing size | Brightness augmentation | 0 | 56/64 | 0.866993 | 16/17 | 13/17 | 1/3/13 | 2/8 |
| HCC missing size | Brightness augmentation | 1 | 59/64 | 0.940850 | 17/17 | 13/17 | 0/4/13 | 0/5 |
| HCC missing size | Brightness augmentation | 2 | 59/64 | 0.917974 | 17/17 | 12/17 | 0/5/12 | 0/5 |
| HCC missing size | Brightness augmentation | 3 | 55/64 | 0.810131 | 16/17 | 10/17 | 1/6/10 | 1/9 |
| HCC missing size | Brightness augmentation | 4 | 60/64 | 0.929412 | 17/17 | 13/17 | 0/4/13 | 0/4 |
| HCC missing size | Gaussian-noise augmentation | 0 | 54/64 | 0.840850 | 16/17 | 11/17 | 1/5/11 | 2/10 |
| HCC missing size | Gaussian-noise augmentation | 1 | 58/64 | 0.917974 | 17/17 | 12/17 | 0/5/12 | 0/6 |
| HCC missing size | Gaussian-noise augmentation | 2 | 58/64 | 0.873856 | 16/17 | 12/17 | 1/4/12 | 1/6 |
| HCC missing size | Gaussian-noise augmentation | 3 | 56/64 | 0.823203 | 15/17 | 11/17 | 2/4/11 | 3/8 |
| HCC missing size | Gaussian-noise augmentation | 4 | 56/64 | 0.886928 | 17/17 | 10/17 | 0/7/10 | 0/8 |
