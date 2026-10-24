# Study log



## 2026-10-22
_Estimated development effort: 2 through today, 167 total_

We made some minor refinements to projections analysis. For the most part the projections were made obselete by the custom caliper work done to generate HCC proxy masks. But that work left one area unexplored: multiple-pass detection rates on small HCC masses (smaller than 2.5cm). Our projections showed 99% accuracy for presence-based detection at 7 passes and 99% accuracy for overlap based detection, IoU>0. We didn't have the data to run the projections to 100%, but that finding is interesting and, while exploraty, we will consider mentioning it in the discussion section of the paper.

Today we set a rough outline for the paper and tentatively setteld on the paper's two primary ideas:

1. Using nnUnet and B-mode ultrasound, it's very cheap to train models to perform triage detection of liver masses.
2. Simply by running multiple passes, such models can get to 100% detection for almost no extra cost.

We'll refine these ideas and work through supporting materials as we arrange the paper; we expect more analysis as we write the paper and will note that here. 


## 2026-10-22
_Estimated development effort: 11 through today, 168 total_

This is a longer entry than most, so we'll summarize it as follows:

### AUL-trained models generalize to external dataset but don't meet our pre-registered 0.03 threshold, failing it for malignant detection and greatly surpassing it for benign, which may result from mass-size discrepancies by pathology. Multiple passes per patient improve the effectiveness of the model for a nominal cost.

Picking up on the last point, we can project that AUL-trained models predict 100% of malignant masses in the SMC dataset after 11 passes.

Given recorded inference speeds of less than 1 prediction per second on a consumer laptop, we consider this one of the studies most important findings of the study. 

It could mean that one of our low-cost models, for a nominal cost in clinical time and energy, could surpass whatever gains might be derived from other methods, such as adding more data, more compute, more complex training pipelines, or novel network architectures. 

At that same 11-pass threshold, exploratory, proxy-mask based prediction rates for overlap IoU>0.0 and Centroid DEQ 1.0 are 99% and 98% respectively.

The caveat there is the size analysis, where HCC detection rates are abysmal at less than 2cm and very good above 8cm. Keeping in mind that sizes analyses themselves are exploratory, given that they are based in the same calipers-derived, perfect-circle, proxy masks.

It's hard to compare these SMC numbers to AUL because, among other discrepancies, the latter doesnt have physical measurements.

---

As for multiple-passes for single patients, we did extract more proxy masks for HCC images from the SMC dataset using calipers and run analysis. The analysis shows more noise than for hemangioma proxy masks, due to (a) the lack of "clean twins" and (b) the heterogenous nature of the acquisition devices and data preparation methods, such as the resizing of all images to a 512x512 standard.

The HCC proxy mask extraction is also problematic in that necessitated both manually standardizing caliper marks, and also creating faux clean twins. The manually created caliper marks, where applied to a subset of images where patients had exactly 4 or more scans. And relatively large subset thereof was discarded accorded to the following rule: only black and white images with exactly two caliper marks were used; the rest (0 marks, 1 mark, 3 marks, 4 marks, doppler etc) were excluded. The maually added marks were used  to create proxy masks for the resulting subset.

Because of the flimsiness of the approach, we continue to regard all analysis derived from proxy masks as exploratory; it is amply reported in the repository, however, and like the presence-based detection rates (which do no need proxy masks), the results partially validate the the AUL-trained models.

With that said, the findings are interesting for exploratory purposes, showing that for multiple passes other triage-level detections views besides presence based improve monotically.

---

Despite the high presence-based detection rates at multiple passes per patient, the external validation experiment against the SMC dataset only partially validates the AUL models, did not fall within our registered 0.03 detection rate range. This metric was consistent with 0.03 detection-rate threshholds registered in advance of other experiments in our study. In retrospect, perhaps it was a mistake to register an AUL-comparitive value as a basis for considering the experiment "validating". We should have defined validating better, since post-hoc, it seems the external validation experiment was valuable in multiple ways. We should have researched the question better and found a more meaningful metric or set of metrics.

Presence-based detection rates were 0.894 for malignant-mass images and 0.924 for benign-mass ones. That's as compared to 0.956 in for malignant in AUL and 0.725 for benign. 

A one-off analysis whereby we purged SMC of color images, made no substantial difference.

It's problematic to compare one dataset's results to another however, without considering size, which as we have seen from regression testing is a significant confound for the performance of the models on the AUL dataset.

In short: it's possible there were proportionally more small malignant masses in SMC than in AUL and more large benign masses in SMC. Caliper-derived proxy masks support this explanation. However, this remains speculative since mass sizes (and locations) were not recorded by segmenation-annotation.



## 2026-10-20
_Estimated development effort: 6 through today, 157 total_

One advantage the external validation set gives us is per patient data where we can see the aggregate detection rate of multiple passes of an ultrasound scan on the same patient.

The numbers seems to confirm near complete patient-level flagging of mass-positive patients with a given number of multiple image acquisitions.

Due to the nature of the SCM data set we can probably only report presence-based detection as confirmatory. 

| Views per patient | Patients | Images | nnU-Net default augmentation |
|---|---:|---:|---|
| 1 | 129 | 129 | 0.916 ± 0.014 |
| 2 | 99 | 198 | 0.988 ± 0.008 |
| 3 | 67 | 201 | 0.988 ± 0.007 |
| 4 | 43 | 172 | 1.000 ± 0.000 |
| 5 | 29 | 145 | 1.000 ± 0.000 |
| 6 | 10 | 60 | 1.000 ± 0.000 |
| 7 | 11 | 77 | 1.000 ± 0.000 |
| 8 | 2 | 16 | 1.000 ± 0.000 |
| 9 | 6 | 54 | 1.000 ± 0.000 |
| 10 | 5 | 50 | 1.000 ± 0.000 |
| 11 | 3 | 33 | 1.000 ± 0.000 |
| 12 | 4 | 48 | 1.000 ± 0.000 |
| 13 | 5 | 65 | 1.000 ± 0.000 |
| 14 | 3 | 42 | 1.000 ± 0.000 |
| 15 | 1 | 15 | 1.000 ± 0.000 |
| 16 | 1 | 16 | 1.000 ± 0.000 |
| 17 | 1 | 17 | 1.000 ± 0.000 |
| 18 | 0 | 0 | NA |
| 19 | 0 | 0 | NA |
| 20 | 0 | 0 | NA |
| 21 | 0 | 0 | NA |
| 22 | 0 | 0 | NA |
| 23 | 1 | 23 | 1.000 ± 0.000 |
| 24 | 0 | 0 | NA |
| 25 | 0 | 0 | NA |
| 26 | 0 | 0 | NA |
| 27 | 0 | 0 | NA |
| 28 | 1 | 28 | 1.000 ± 0.000 |

Custom augmentations perform nearly as well, with only Gaussian noise failing to achieve perfect detection at 10 passes (though also perfect from 4-9 and 11-28).

For hemangioma, thanks to the proxy masks we extracted from caliper annotations, we have similar-shaped metrics for other views of detection.

We will  try to derive more proxy masks from the SMC dataset in order to apply the analysis above to other forms of detection besides presence-based. Failing that, or supporting it, we'll also do a beta binomial analysis, to firm up projections. 

We view metrics derived from such proxy masks as exploratory not confirmatory but worth documenting.

-----

Our paper trail budget appears to come to approximately $91. We'll reverify that before reporting. The vast majority of that budget went to model training - 40 models were trained in the main portion of the study.


| Experiment | Models trained | final | best | milestone checkpoints |
|---|---|---|---|---|
| 01 training efficiency (5 folds × 1,000 epochs) | 5 | 5 | 5 | 40 (8/fold: 25, 50, 75, 100, 150, 300, 500, 750) |
| 02 data efficiency (4 sizes × 5 folds) | 20 | 20 | 20 | 100 (5/model: 5, 25, 75, 100, 125) |
| 03 augmentation ablation (2 sizes × 5 folds) | 10 | 10 | 10 | 0 |
| 04 gaussian noise ablation (1 size × 5 folds) | 5 | 5 | 5 | 0 |
| **Total** | **40** | **40** | **40** | **140** |

Total saved checkpoints: 220 (excluding dry run checkpoints)



## 2026-10-16
_Estimated development effort: 2 through today, 153 total_

The external validation results confirm that the study's efficiency baselines are applicable to an independent external dataset.

We think this provides a foundation for subsequent research into  efficient ultrasound  malignant-mass detection. Thus it's a fitting point to halt the research phase of the project and proceed with writing the paper. We may do futher analysis; we dont anticipate conducting further research.


## 2026-10-16
_Estimated development effort: 6 through today, 151 total_


We designed the final experiment, external validation against the SMC-LUD dataset. After initially running it locally, we found it would take 170 hours to complete, and decided to run it on a GPU, in order to be able to move forward with the study more quickly.

In reviewing the dataset, we found de facto annotation, size annotation and location annotation, in the form of calipers. Calipers are not as rigorous as segmentation annotation; they cannot be used for segmentation analysis. But they are reliable enough that we can do analysis of overlap detection views. This 
represent a slight change in direction; we'd previously anticipated only being able to measure the performance of presence-based detection. We'll treat metrics based on overlap-based detections as exploratory rather than definitive.

To facilitate this approach, we wrote a "caliper extraction" script that reads caliper marks to figure out where the lesion is and how large it is, then writes a proxy reference mask.

It does this in a different way for the HCC images than for the Hemangioma images. Furthermore, these are really two different sets of images with respect to dimensions and possibly quality. We won't directly compare results from pathology type to the other.

Before running the experiment, we'ee going to register a the 0.03 triage detection-rate threshold for relevance. That is, we'll consider the results of the experiment validating if they they come within 0.03 points for a given triage detection rate - presence-based, overlap IoU>0, overlap IoU>0.2. 

We're going to attempt to validate 3 5-fold sets of models:

- 588-images at 150 epochs, from experiment 02
- 588-images at 150 epochs with increased brightness frequency, from experiment 03
- 588-images at 150 epochs with Gaussian noise eliminated, from experiment 04




## 2026-10-15
_Estimated development effort: 5 through today, 145 total_

We performed significant mass-size analysis to determine whether the difference in segmentation and detection rates from benign to malignant was down to size. 

The answer seems to be mostly down to size, partly down to pathology, and a regression analysis reveals confidence intervals taht are too wide too firm up that statement.

Looking masses by quartile, we see malignant-mass detection rates of 100% on several views of detection for the largest quartile. Yet benign mass detection never rises to 100%.

Binning masses logarithmically tells a similar story but indicates that size plays an important role, with both benign and malignant-mass detection hitting 100% for the largest bin of mass sizes accross multiple views of detection.

[Majority detection by annotated mass area](majority_detection_by_size.png)


Most relevant to this study is how well malignant-mass detection holds up at small sizes, with malignant masses detection rates remaining high (78 to 91%) for two smallest mass-size bins.


## 2026-10-14
_Estimated development effort: 3 through today, 140 total_

With respect to malignant-mass detection, the effect of the experiment in eliminating Gaussian noise augmentation had mixed results.

As with the brightness frequency experiment, there are signs that adjusting nnU-Net's defaults could pay off. There's not clear-cut win (along the lines of dropping epoch budget from 1000 to 150) though that we encountered. It seems likely that find the pay off would take extensive experimentation, including testing variations, and probably falls outside the scope of this study.

Were the subject to be explored, we'd want to test brightness multiplication, potentially pair that with frequency. We also want to test removing or reducing Gaussian blue and pair that with Gaussian noise. And then test pairs of those pairs, and on.

What we do see are modest improvements related to both ablations. We'll test both in our external validation experiment, along with the control. 



## 2026-10-14
_Estimated development effort: 9 through today, 137 total_

Having reviewed and analyzed the brightness-frequency augmentation ablation tests, we have decided to pursue an additional augmentation ablation: removing Gaussian noise. Tupper & Gagné found Gaussian noise to be the most significant augmentation factor on ultrasound.

Of the many other augmentation tests that make sense for B-mode ultrasound, this seems like the lowest hanging fruit. The results from the prior ablation justify it as an area of exploration, before moving on to the planned efficiency experiment and external validation testing.

As with the prior augmentation ablation, we will register our assessment of the results around the 0.03 mark: detection improvements of 0.03 or better are significant, as are false positives decreases of 0.03 or better. 

Without forecasting every permutation, a preponderance of such results answers the question "is removing Gaussian noise on B-mode ultrasound for the purpose of improving malignant-mass detection a worthwhile, cost-efficient methodological approach?" with "probably"

---


Looking at the results of the augmentation ablation, where we asked: "will modifying this nnU-Net brightness-augmentation default improve malignant-mass detection rates and/or lower false positive rates?" 

We found that for the 588 image-pool models, the answer is between `maybe` or `probably not`, with little or no improvements in detection rates and a modest improvement in false positives.

For the the 588 models final checkpoint, the false positives rate for malignant triage went from 0.15 to 0.12, a solid 0.03 drop. For the best checkpoint, it dropped from 0.19 to 0.13, a 0.06 drop paired - but both improvements comes against a 0.002 drop in detection rate (essentially null). That's a net improvement if not a dramatic one to suggest that modifying augmentation could be worth exploring.

For the other detection metrics and 588 image models, we saw a similar pattern but less pronounced, with slight decreases in detection rates and slightly larger decreases in corresponding false positive rates.

For the ioU>0.2 metric on the final checkpoint, we saw a 0.005 improvement in detection paired with a 0.008 decrease in false positives, putting it in the `maybe` category. 

For the 74 image models, the effect of the augmentation was slighly more pronounced, with modest improvements in detection paired with concerningly large increases in corresponding false positive rates. For example, for the final checkpoint, the malignant triage detection rate increased by 0.01, but false positives on normal images increased by an alarming 0.1.

For the best checkpoint at the IoU>0.2 metric there was a 0.016 increase paired with a .005 increase in false positive rate, but that was not the trend. For the 74 image models, this augmentation modification is `maybe` useful.

On the most permissive triage metric (presence-based detection), the highest performing single model from experiment 02 was the 588 image, best checkpoint on fold4 — with detection of 0.966, and false positives rate of 0.05.

Looking at it from the top-level point of view, we do see a slight improvements in some views of triage-level detection, particularly presence based detection, which is the metric that will evaluated during the external validation experiment. For that experiment, we'll compare the 74 and 588 models from both 02 and 03, to see how they perform in predicting malignant masses.

#### Presence-based (triage)

| Experiment | Detection | FP | Blended |
|---|---|---|---|
| 02 data_efficiency  | **0.956 ± 0.010** | 0.150 ± 0.100 | 0.806 ± 0.103 |
| 03 augmentation_ablation  | 0.954 ± 0.008 | **0.120 ± 0.027** | **0.834 ± 0.023** |

#### IoU > 0

| Experiment | Detection | FP | Blended |
|---|---|---|---|
| 02 data_efficiency  | **0.940 ± 0.015** | **0.107 ± 0.038** | **0.833 ± 0.042** |
| 03 augmentation_ablation  | 0.931 ± 0.011 | 0.110 ± 0.012 | 0.821 ± 0.013 |

#### IoU > 0.2

| Experiment | Detection | FP | Blended |
|---|---|---|---|
| 02 data_efficiency  | 0.899 ± 0.015 | 0.177 ± 0.043 | 0.722 ± 0.055 |
| 03 augmentation_ablation  | **0.903 ± 0.006** | **0.169 ± 0.021** | **0.735 ± 0.022** |

#### Centroid d_eq (100%)

| Experiment | Detection | FP | Blended |
|---|---|---|---|
| 02 data_efficiency  | **0.936 ± 0.010** | **0.105 ± 0.035** | **0.831 ± 0.040** |
| 03 augmentation_ablation  | 0.922 ± 0.005 | 0.109 ± 0.019 | 0.813 ± 0.021 |

Altogether, the results indicate that overriding nnU-Net augmentation defaults will probably make a difference for malignant mass detection on ultrasound.  What's uncertain is how to orchestrate those over-rides, with (reducing) Gaussian noise and increasing brightness multiplication being potential areas of exploration.

---

Exploratorily, we looked at internal epoch milestones during the data efficiency training too. As with the original training efficiency study, there were interesting results at 75 epochs, especially against the 147 image training set size, with malignant detection of 0.86 for IoU>0 and 89 for presence-based detection.

| Training pool | Presence-based detection | Malignant detection, IoU >0 | Malignant detection, IoU >0.2 | FP/image |
|---:|---:|---:|---:|---:|
| 74 | 89.2% → 90.8% | 81.6% → 85.1% | 56.6% → 60.7% | 0.427 → 0.429 |
| 147 | 89.4% → 88.0% | 86.0% → 83.9% | 73.8% → 74.7% | 0.290 → 0.245 |
| 294 | 89.7% → 91.5% | 85.7% → 89.0% | 80.0% → 80.5% | 0.162 → 0.152 |
| 588 | 93.3% → 95.6% | 91.7% → 94.0% | 89.0% → 89.9% | 0.125 → 0.107 |

This table shows the change from 75 epochs to 150.

## 2026-10-13

The ablation experiment asks whether modifying nnU-Net augmentation defaults is a low-cost way to improve malignant mass detection rates while also lowering false positive rates. For efficiency's sake the scope of augmentation is limited to brightness, inspired by (though not replicating) Tupper & Gagné's [Revisiting Data Augmentation](https://arxiv.org/html/2501.13193v2) for U-Net segmentation on the same AUL dataset (among others). Specifically, the ablation will set the frequency of brightness augmentation to 0.5, as opposed to the nnU-Net default of 0.15. No other modification will be made to brightness or any other form of augmentation.

The refined question becomes: **will modifying this nnU-Net brightness-augmentation default improve malignant-mass detection rates and/or lower false positive rates?**

For us, the answer will be `probably` if two of these three views of detection are improved by more than 0.03:

- presence-based detection
- iou>0.0 detection
- iou>0.2 detection

0.03 was the range used to set the epoch budget of 150 during experiment 01 on training efficiency.

And if false positives decrease by more than 0.03 for those same views of detection.

The answer will be `maybe` if detection or false positive metrics are better by between 0.01 and 0.03. 

It will be `probably not` if changes in both metrics are under 0.01.

If either detection or the false positive metric is substantially bettered (more than 0.05), the answer to the question above will also be `yes`.

If results are muddied by being better in one respect and worse in another, for a varying number of models, we'll evaluate on a case by case basis; it would be a waste of effort to walk through and pre-register every possible permutation.

The ablation experiment will be applied to the largest (588) and smallest (74) models trained during experiment 02.

Other augmentation tests could potentially impact malignant mass detection on ultrasound (eg reducing Gaussian noise) but here we focus on just one augmentation transform for the sake of efficiency.



## 2026-10-12
_Estimated development effort: 2 through today, 128 total_

The results of the data efficiency experiment show that smaller dataset models perform well in most views of malignant mass detection, but that smaller models perform badly with respect to triggering false alarms.




## 2026-10-11
_Estimated development effort: 6 through today, 126 total_

Because Verda, as with many GPU cloud providers, frequently updates pricing and has varied availability, we chose to update the runbooks to be device agnostic. The initial dry runs and the training efficiency experiments ran on an older RTX A6000. Experiment 02, however, runs on an RTX 6000 Pro. The total cost should come to within 10% more of what was anticipated.

Furthermore, Verda discontinued the ability to use a vanilla Ubuntu VM with the A6000; it only allows Ubuntu through a Docker image. A triviality ultimately, but before running 02, we ran a 3rd dry run against a Verda Docker image and made a series of updates to the runbooks accordingly.

Ultimately, we will provide "one-click" script to run the entirely easily, in the interest of reproduction. But the runbooks are meant to help with troubleshooting and also with understanding of how the configuration is designed.

An unanticipated block storage issue caused a mid-run failure and stoppage for hte 02 data efficiency experiment. We remedied the stoppage.



## 2026-10-06
_Estimated development effort: 11 through today, 120 total_

Having completed the training efficiency experiment and evaluated the results according the to criteria registed in this study, on September 28, we have found 150 epochs to be the most cost-efficient training budget for triage-level detection of cancerous masses on AUL data.

At 150 epochs, notwithstanding the disadvantage of learning rate, models across the 5 folds had the 2nd highest malignant-mass detection rate for the overlap-based detection metric, where IoU>0.

The epoch budget with the highest malignant-mass detection rate was, interestingly, not 1000 but 300.

In addition to performing analysis, we implemented the predictions dataset data layer, as planned. It holds a custom calculated false positive rate for each detection metric used in the study.

We conducted a noise floor analysis across 6350 prediction in the training efficiency experiment that confirmed 0.03 as the ideal relative noise floor - it's the largest two-digit fraction to cause 0 predictions to be missed across all 7 detection metrics tracked in the study.



## 2026-10-05
_Estimated development effort: 3 today, 109 total_

As the efficiency training came to a close, it gave us a chance to polish the runbooks and reproducability and lets us get started on analysis.

The first order of business with respect to analysis will be to create some kind of data layer in between the actual predictions results and checkpoints, archived in `archives/*` and the analysis code. This should make analysis scripts run more efficiently and also create a single point of data transformation.

Over the weekend, we also put some thought into an abalation experiment after looking more closely at a prior study evaluating the effect of augmentation techniques on mass segmentation for the AUL dataset (Tupper & Gagné 2024). Specifically, they found that increasing brightness increased mass segmentation Dice by 0.02. They also found that Gaussian blur actually worsened segmentation. Therefore we think it would suit our interest in efficiency to evaluate whether such adjustments would benefit an nnU-Net pipeline with the same data set.

In fact, before running the training efficiency experiment, we tried to align our study design where possible. We're already using the same 588 split, but we also chose to firstly set aside the very Ultrabench-based test set that Tupper & Gagné used. This will not allow 1:1 comparisons (nor is that a desired outcome necesarily) because there remain a broad number of differences in methods.

In any case, we believe there are specific findings that we could repurpose in an ablation studying asking: "Can adjusting nnU-Net augmentation defaults such as brightness results increase triage-level detection rates and lower corresponding false positive rates?".




## 2026-10-02
_Estimated development effort: 3 today, 106 total_

After simplifying the training efficiency runbooks and bringing the others in line, we ran a number of tests of the dry run, each costing about 0.20 cents on spot instances.

On the fourth try once we were satisfied with the runbook, we decided to keep using the same instance for the full run of the training efficiency experiment.

At the same time, we noticed a couple of things about Verda. Firstly, fairly significant price hikes accross the board. Not surprising given their ongoing high demand - most instance types are unavailable most of the time. Secondly we noticed a new model on the lower end, the RTX A6000, the predecessor to the RTX A6000 we had thought we would use for the study. The RTX A6000 is quite a bit slower but is very competitively priced and the preliminary dry run studies we did indicated that it would be more cost efficient overall, so we have decided to standardize on it for the duration of the study.

The price changes prompted us to re-review competing options, again with these criteria:
- Entirely EU based (headquarted in EU, fully owned by EU interests, no datacenters outside the EU)
- Flexible and well supported API
- Mission-driven
- Low-cost options
- Good user experience

Based on the above we chose to continue using Verda.

A small snafu though - we were kicked off of the instance mid-run. The data however is preseverved so that when we create a new instance to resume the training run, we can pick up where we left off. Mending the logs will require some custom work not detailed in the runbooks.

Making an idempotent bootstrap script for setup is on our list of todo's for the second experiment - the runbook becomes a reference and troubleshooting tool. We'll make that for the subsequents experiments and backport it to this one for this wanting to reproduce the study. This will have the added advantage of cost efficiency.


## 2026-10-01
_Estimated development effort: 8 today, 103 total_

Refining the runbooks for usability since reproducibility is a key component of cognitive effiency related to the study's baselines.


## 2026-09-30
_Estimated development effort: 7 today, 95 total_

Working on scripting the initial experiment today and mapping out what gets logged, how data is prepared and converted, how training is conducted and how predictions are run.

We've created experiment runbooks that take users through instance preparation and a systematic way to verify results and download them before finishing an instance.

We've created an optional cheap dry run for the first experiment that can be used to test the runbook in a potentially differing GPU environment.

WE also created an external remote control that can be run from any CPU, including a hosted cloud CPU that automatically deletes a rented Verda.com instance after downloading its results.

The focus has been on reproducability both from a technical and cognistive perspective. 

Well run the dry run to verify the data conversion and prep scripts, runbooks, runners, and custom trainers, and a custom inference logging script.

Renamed the 3rd experiment from inference_efficiency to prediction_efficiency so it's intelligible to a broader audience.



### Morning

Organizing the layout of the study to be intuitive visually and support the narrative of the study as follows:

`
analysis/
  training_efficiency/
  data_efficiency/
  inference_efficiency/
  external_testing/
  study_reports/
experiment_execution/
  training_efficiency/
    custom_trainers/
  data_efficiency/
    custom_trainers/  
  inference_efficiency/
    custom_trainers/  
  external_testing/
    custom_trainers/  
logs/
  training_efficiency/
  data_efficiency/
  inference_efficiency/
  external_testing/`






## 2026-09-28
_Estimated development effort: 8 today, 88 total_

### Morning

#### Data scaling experiment

The data scaling experiment will train models at various training-set sizes across the nnUnet default of 5 folds. Thus the study will use an 80/20 train test-split (588/147).

The planned data scales are 100%, 50%, 25%, and 12.5%: 588 images, 294 images, 147 images, and 74 images. 

Smaller sizes will be nested subsets of larger ones and keep the proportions of AUL:

- 435 malignant (59.2%)
- 200 benign (27.2%)
- 100 normal (13.6%)

We may add a smaller 6.25% (37 images) size if it seems valuable.

The 588/147 split was also used on AUL in Tupper & Gagné's 2025 augmentation study, letting us ground results in prior research and make comparisons.

The training budget of the data scaling experiment will be determined by a preliminary experiment.

#### Detection metrics

The study will measure:
- presence-based detection 
- overlap based detection with an IoU>0.0 
- overlap based detection with an IoU>0.2
- overlap based detection with an IoU>0.5
- centroid within 0.25x GT equivalent diameter
- centroid within 0.5x GT equivalent diameter
- centroid within 1.0x GT equivalent diameter

While the study will evaluate detection of both malignant and benign masses, the focus is on malignant masses. 

Of these 7, our working assumption is that these three to be most closely correlated to triage-level detection:

- presence-based detection 
- overlap based detection with an IoU>0.0 
- centroid within 1.0x GT equivalent diameter

Of these, we consider overlap-based detection with an IoU>0.0 to be the dedicated triage metric, as the most permissive metric to incorporate localization and second most permissive triage-level metric overall


#### Training budget experiment

The training budget experiment will train on all 588 images to the nnUnet default of 1000 epochs across five folds, while saving internal checkpoints at these epochs: 25, 50, 75, 100, 150, 300, 500, 750. 

Each internal checkpoint will be compared against one another and against the final budget. 

Training budget selection will account for three factors: 
- Detection rate of the overlap IoU>0.0 detection metric on malignant masses*
- Training cost; larger training costs will be penalized
- Learning rate. The greater the learning rate of a checkpoint, the worse it can perform at detection

Today we'll register the selection criteria as follows: the winning budget is the smallest one across seeds whose detection rates comes within a 0.03 of the highest one.

Why not train separate runs at varying epoch budgets to compare fully annealed final epochs against one another?  Accounting for learning rate discrepancies nets us a comparable and more efficient methodology. In any case, either way incorporates a gut call on the value of training cost in determining the winner.


#### False positives

False positives are important but will suffer from lack of data. The 588-image training-set will include just 80 normals and 74-image training-set 10. For both, the held out false positives will be 20, giving us a fairly large confidence interval.

Still we will evaluate each detection metric against its corresponding false positives.

Because the AUL dataset has exactly one patient per image, all false positive rates are effectively case-level, as opposed to lesion level.


#### Noise floor

To elimate speckle-driven small predictions, we will use the training budget experiment to set a noise floor that is as high as possible without causing malignant-mass detection misses across all 7 metrics against the 147-image held out test set. 

Because AUL has no physical calibration, and because its images vary greatly in size and composition, we'll use a relative metric (against the total_ image size). We'll test the following noise-floor sizes: 0.0, 0.005, 0.01,  0.015, 0.02,  0.025, 0.03,  0.035, 0.04, 0.045, and 0.05.

#### External validation

After concluding the training budget experiment, data scaling experiment, CPU predictions experiment, and any ablations studies, we'll also perform an external validation experiment against another liver ultrasound dataset: SMC-LUD. SMC-LUD contains 5,385  2D B-mode liver ultrasound images from 1,021 patients - 2,716 are HCC and the other 2,669 are hemangioma. It has no normals and no  segmentation annotations, only pathology classification. This will allow us to measure:

- Malignant-mass detection rate
- Benign-mass detection rate
- Combined-mass detection rate

Because we don't have annotations in SMC-LUD, the only detection view we can assess is presence-based detection.

#### Network architecture

We will run the study on PlainConvUNet 2D, as opposed to the new nnUNet architecture Resenc. Most literature and published benchmarks about Resenc concern 3D. Resenc M 3D, the lowest-cost tier of Resenc in terms of compute, appears to have a small performance edge over PlainConv in large sizes. There's no evidence one way or the other of a performance benefit with either 2D or with smaller training size models. Meanwhile, there's direct evidence that Resenc incurs greater inference cost (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md); thus it is off the table for this study.



## 2026-09-27
_Estimated development effort: 8 today, 88 total_

The larger interest motivating the study is affordable, accessible cancer screening and monitoring. This study focuses that interest on liver cancer, a major and growing concern in many regions of the world. 

The study's primary goal is to establish resource-efficiency baselines for training AI to perform triage-level detection of malignant masses in the liver. Implicit in 'triage-level detection' is detecting relatively small masses, let's say below 2 centimeters, corresponding to early stage detection (Reig et al 2026). The secondary goal is to establish corresponding baselines for monitoring.

Efficiency means accessible and low-cost in several ways:

- Training cost, primarily in terms of GPU compute; can it be trained cheaply - where and how exactly?
- Development-effort cost: can the needed research, planning, design, tuning, setup, and execution - of the entire training pipeline - fit into a modest budget?
- Data acquisition cost: can detection AI be trained and tested on a realistic quantity of labelled and annotated imagery?
- Cognition cost: can an AI detection training project be broadly understood by a range of stakeholders, including those without clinical or ML expertise
- Verification cost: how easy and affordable is it to reproduce the entire study and thereby verify its results? 
- Deployment cost. Does it fit on consumer-grade laptops and phones; can it run inference on such devices cheaply, just like in a real-world patient care setting?

To address many of these concerns, the imaging modality must be ultrasound, the lowest-cost and most widely available form of medical imaging. Furthermore, it means B-mode ultrasound, as on portable, handheld "POCUS" ultrasound devices. No suitable POCUS-imagery dataset being publicly available, however, the study's training corpus will be the Annotated Liver Ultrasound (AUL) images dataset. AUL consists of a mix of malignant-mass, benign-mass and normal livers on static B-Mode images. This dataset has been been used in several published studies though none concerned with efficiency.

On the technology side, the deep learning network architecture used to train models on B-mode ultrasound must be open source, relatively simple, segmentation-based, and 2D-capable. Factoring in the importance of development-effort efficiency as well, the study will adopt the efficient, self-configuring nnU-Net framework, using its PlainConvUNet 2D network architecture. The study will leverage and adhere to nnU-Net defaults, except when in conflict with the goal of efficiency. For example, the study will try to establish more economical epoch budgets than nnU-Net's 1000 epoch default. But there will be no network-architectural modifications whatsoever.

The study will evaluate the performance of trained models across a broad range of detection metrics meant to correlate to either triage screening or monitoring clinical uses cases (or both). We will also report and provide analysis on segmentation. Many of the 8 metrics we'll report could be proxies for triage screening and have been used as such in published studies. Others may better correlate to monitoring. Ultimately those are clinical distinctions that the study will leave up to readers.

To address the common concern of the limited availability of labelled data, the study will report and evaluate results across a sweep of training-dataset sizes. Other analysis will include determining optimal training length (epoch budget), investigating ideal noise floors for ultrasound speckle, detailing model footprints, and wall-clock measuring GPU and CPU performance on training and especially on inference. These may call for separate experiments within the study.

_Estimated development effort: 80, including health, clinical, and technical research and experimentation preceding and inclusive of this log entry
