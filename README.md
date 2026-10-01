# Low-cost baselines for detecting liver malignancies on ultrasound

This study presents low-cost, resource-efficient baselines for cancerous liver-mass detection on B-mode ultrasound using nnU-Net.

The study is motivated by the need to lower costs in the following areas of cancer-detection AI: 

- Training data acquisition cost
- Development-effort cost
- Model-training cost
- Model-deployment cost
- Study comprehension cost
- Study verification cost

Each cost is addressed in the study, either by direct experiments or study methodology.

This study trains 2D PlainConvUNet on the Annotated Ultrasound Liver images [dataset](https://doi.org/10.5281/zenodo.7272660) and establishes malignant-mass detection baselines across training budgets and training-set sizes;  prediction  / inference benchmarks are tested on GPU and consumer devices.

The [study log](STUDY_LOG.md) documents progress and development hours.

**License**: [Apache 2.0](LICENSE.md).
