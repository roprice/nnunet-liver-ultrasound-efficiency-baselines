# Low-cost baselines for detecting liver malignancies on ultrasound

This study presents low-cost, resource-efficient baselines for cancerous liver-mass detection on B-mode ultrasound using nnU-Net.

The study is motivated by the need for affordable, accessible liver-cancer screening. Thus, it is in turn motivated by exploring cost-efficiencis in these areas of cancer-detection AI: 

- Training data acquisition cost
- Development-effort cost
- Model-training cost
- Model-deployment cost
- Study reproducablity cost

Each cost is addressed in the study, either by direct experiments or study methodology.

This study trains 2D PlainConvUNet on the Annotated Ultrasound Liver images (https://doi.org/10.5281/zenodo.7272660) and establishes malignant-mass detection baselines across training budgets and training-set sizes;  prediction  / inference benchmarks are tested on GPU and consumer devices. 

Ablation studies examine whether deviating from nnU-Net augmentation defaults affect detection rates at a relatively low cost.

External testing against the SMC-LUD dataset assess whether the models trained on AUL work elsewhere. Because SMC-LUD has multiple images per patient (unlike AUL), testing against it helps assess whether there are low-cost detection gains from getting multiple images from a patient at point of care.

The study log documents progress, registers predictions of experiment outcomes, and tracks development effort in hours.
