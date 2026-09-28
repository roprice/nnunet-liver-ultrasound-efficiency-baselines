## Study log



## 2026-09-27

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

To address the common concern of the limited availability of labelled data, the study will report and evaluate results across a sweep of training-dataset sizes. Other analysis will include determining optimal training length (epoch budget), investigating ideal noise floors for ultrasound speckle, detailing model footprints, and wall-clock measuring GPU and CPU performance on training and especially on inference.

Estimated hours of research and experimentation preceding and inclusive of this log entry: 80.
