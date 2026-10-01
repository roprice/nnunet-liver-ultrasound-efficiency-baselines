This section contains the code used to run the training pipeline. 

Each experiment has its own set of code, including a markdown runbook for step by step reproduction, allowing for environment issues to be pinpointed.

These files are shared among experiments:
- benchmark_inference.py
- download_archive.sh (downloads/checksum-verifies a run from the GPU  to `archives/<experiment>`)
- prepare_data/aul_conversion.py
- prepare_data/aul_splits.py 
- prepare_data/aul_splits_verification.py 

A training-efficiency dry run experiment lives in `01_training_efficiency/dry_run/` and lets you diagnose any issues before starting a full run.
