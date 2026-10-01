This section contains the code used to run the training using nnUnet. Each experiment has its own set of code, including a markdown runbook.

These files are shared:
- benchmark_inference.py
- prepare_data/aul_conversion.py
- prepare_data/aul_splits.py (shared train/test split, nested subsets, and five-fold assignments)
- prepare_data/aul_splits_verification.py (split tests; run with `python3 -m unittest discover -s experiments/prepare_data -p '*_verification.py'`)
