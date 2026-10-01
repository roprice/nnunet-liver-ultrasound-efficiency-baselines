This section contains the code used to run the training using nnUnet. Each experiment has its own set of code, including a markdown runbook.

The experiments involving compute have an optional Verda.com-specific script for automating the termination of the runs, to save on GPU rental costs.

There's also a faux experiment called training efficiency dry run - use it to test how well the runbook for training efficiency  and subsequent experiments maps to whatever environment youre reproducing the study in.

These files are shared:
- prepare_data/aul_conversion.py (requires `--reference-mapping prepare_data/reference/case_mapping.json`, the committed seed-42 assignments)
- prepare_data/aul_splits.py (shared train/test split, nested subsets, and five-fold assignments)
- prepare_data/aul_splits_verification.py (split tests; run with `python3 -m unittest discover -s experiments/prepare_data -p '*_verification.py'`)
- benchmark_inference.py
