# Study log

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
