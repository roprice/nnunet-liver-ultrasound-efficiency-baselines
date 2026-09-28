## Study log



## 2026-09-27

The larger interest motivating the study is affordable, accessible cancer screening and monitoring. That interest will be bounded in this study by liver cancer in particular, a major and growing concern in many regions of the world. 

To that end, the primary goal of this study is to establish resource-efficiency baselines for training AI to perform triage-level detection of malignant masses in the liver. Implicit in 'triage-level detection' is detecting relatively small masses, let's say below 2 centimeters, corresponding to early stage detection (Reig et al 2026). The secondary goal is to establish corresponding baselines for monitoring.

The baselines don't need to, and probably won't, establish clinical value. The will establish research value into the efficient creation of malignant liver-mass detection AI. 

Efficiency means accessible and low-cost in these ways:

- Training cost, primarily in terms of GPU compute; can it be trained cheaply - where and how exactly?
- Development-effort cost: can the needed research, planning, design, tuning, setup, and execution - of the entire training pipeline - fit into a modest budget?
- Data acquisition cost: can detection AI be trained and tested on a realistic quantity of labelled and annotated imagery?
- Cognition cost: can an AI detection training project be broadly understood by a range of stakeholders, including those without clinical or ML expertise
- Verification cost: how easy and affordable is it to reproduce the entire study and thereby verify its results? 
- Deployment cost. Does it fit on consumer-grade laptops and phones; can it run inference on such devices cheaply, just like in a real-world patient care setting?

To meet these concerns, the imaging modality must be ultrasound, the lowest-cost and most widely available form of medical imaging. Furthermore, it means B-mode ultrasound, as on portable, handheld "POCUS" ultrasound devices. No suitable POCUS-imagery dataset being publicly available, however, the study's training corpus will be the Annotated Liver Ultrasound (AUL) images dataset. AUL consists of a mix of malignant-mass, benign-mass and normal livers on static B-Mode images. This dataset has been been used in several published studies though none concerned with efficiency.

On the technology side, the deep learning network architecture used to train models on B-mode ultrasound must be open source, relatively simple, segmentation-based, and 2D-capable. Based on those premises, the study will use a simple 2D U-Net. And based on the importance of development effort efficiency, the study will adopt the nnU-Net framework, using its PlainConvUNet 2D network architecture. The study will leverage and adhere to nnU-Net defaults, except when in conflict with the goal of efficiency. For example, the study will try to establish more economical epoch budgets than nnU-Net's 1000 epoch default. But there will be no network-architectural modifications whatsoever.

The study will evaluate the performance of trained models across a broad range of detection metrics meant to correlate to either triage screening or monitoring clinical uses cases (or both). We will also report and provide analysis on segmentation. 5 of the 8 metrics we'll report are meant to correlate to triage screening. The rest are meant to correlate to monitoring. Ultimately however, those are clinical distinctions that the study will leave up to readers.

To provide efficiency insight addressing the commonly cited concern of the limited availability of labelled data, the study will determine optimal training (epoch budget) against baselines to be specified later, report results across a sweep of data scales, detail model footprints, and report both GPU and CPU performance on training and, in particular, inference.

Estimated hours of research and experimentation preceding and inclusive of this log entry: 80.
