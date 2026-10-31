# Low-cost baselines for detecting liver malignancies on ultrasound

This study presents low-cost, resource-efficient baselines for cancerous liver-mass detection on B-mode ultrasound using nnU-Net.

The study is motivated by the need for affordable, accessible liver-cancer screening. It is explores cost-efficiencies in these areas of cancer-detection AI: 

- Training data acquisition cost
- Development-effort cost
- Model-training cost
- Model-deployment cost
- Study reproducablity cost

Each cost is addressed in the study to varying degrees, either by direct experiments or study methodology.

This study trains 2D PlainConvUNet on the Annotated Ultrasound Liver images (https://doi.org/10.5281/zenodo.7272660) and establishes malignant-mass detection baselines across training budgets and training-set sizes;  prediction  / inference benchmarks are tested on GPU and consumer devices. 

Ablation studies examine whether deviating from nnU-Net augmentation defaults affect detection rates at a relatively low cost.

External testing against the SMC-LUD dataset assess whether the AUL-trained models are "generalizable", ie whether they work elsewhere. Because SMC-LUD has multiple images per patient (unlike AUL), testing against it helps assess whether there are low-cost detection gains from acquiring multiple images for a given patient. Its physical measurements, which AUL lacks, allow us to report performance on small malignant masses.

The study log documents progress, registers predictions of experiment outcomes, and tracks development effort in hours.
