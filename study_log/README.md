## Study log



## 2026-09-27

The larger interest motivating the study is low-cost, small-footprint, accessible cancer screening and monitoring. That interest will be bounded in this study by liver cancer in particular, a major and growing concern in many regions of the world. 

To that end, the primary goal of this study is to establish efficient baselines for training AI to perform triage-level detection of malignant masses in the liver. Implied in the term triage-level detection is an interest in detecting smaller masses below 2 centimeters. The secondary goal is to establish corresponding baselines for monitoring.

The baselines don't need to establish clinical value as long as they establish research value. Efficiency doesn't just mean compute, though that's part of it. It means low-cost and broadly accessible in every dimension: 
- low development effort: reasonable to plan, research, design, tune, setup, and run
- low cost to train, primarily in terms of GPU compute
- low-cognitive cost: easily reproduced and understood, even for those without clinical or machine learning expertise
- low cost to deploy and run. Can it run inference cheaply on consumer hardware?

For the reasons above, the imaging modality must be ultrasound, the lowest-cost and most broadly available form of medical imaging. Furthermore, low quality, B-mode ultrasound, loosely corresponding to cheaper and more portable handheld "POCUS" ultrasound devices is preferable. The study will use the Annotated Liver Ultrasound (AUL) images dataset as the training corpus.

On the technology side, the deep learning network architecture used to train models on B-mode ultrasound must be open source, relatively simple, segmentation based, and 2D-capable. Based on those premises, the study will use a simple 2D U-Net. And based on the importance of relatively low development effort, the study will adopt the U-Net framework nnU-Net, using its PlainConvUNet 2D network architecture. The study will leverage and adhere to nnU-Net defaults, except when in conflict with the over-arching goal of efficiency.

To provide research value, the study will evaluate the performance of trained models across a broad range of detection metrics meant to correlate to either triage screening or monitoring clinical uses cases (or both). We will also report and provide analysis on segmentation. Our expectation is that 5 of the 8 metrics we report will correlate to the study's primary use case concern of triage screening, with the other 3 better correlated to monitoring. 

That distinction will be left to readers however, and the study will be too data-limited in any case to establish clinical relevance; the goal is research-relevant baselines.

To provide efficiency insight, the study will determine optimal training (epoch budget) against baselines to be specified later, report results across a sweep of data scales, detail model footprints, and report both GPU and CPU performance on training and, in particular, inference.

Estimated hours of research and experimentation preceding and inclusive of this log entry: 80.
